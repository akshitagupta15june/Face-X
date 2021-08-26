using Newtonsoft.Json;
using ServiceResult;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Xamarin.Essentials;
using TokenResponse = SuaveKeys.SnapReader.Uwp.Models.TokenResponse;
namespace SuaveKeys.SnapReader.Uwp.Services
{
    public class SuaveKeysService : ISuaveKeysService
    {
        private readonly ISuaveKeysAuthSettings _authSettings;
        private string _currentToken;
        private const string TokenInfoKey = "SUAVE_TOKEN_INFO";
        private string _baseUrl = "https://suavekeys-dev.azurewebsites.net";
        private readonly string _sessionId = Guid.NewGuid().ToString();
        public SuaveKeysService(ISuaveKeysAuthSettings authSettings)
        {
            _authSettings = authSettings;
        }

        public async Task<Result<string>> GetAccessTokenAsync()
        {
            try
            {
                if (!string.IsNullOrEmpty(_currentToken))
                    return new SuccessResult<string>(_currentToken);

                var refreshResult = await RefreshToken();
                if (refreshResult?.ResultType == ResultType.Ok)
                    return new SuccessResult<string>(_currentToken);

                return new InvalidResult<string>(refreshResult.Errors?.FirstOrDefault());

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
                return new UnexpectedResult<string>();
            }
        }

        private async Task<Result<bool>> RefreshToken()
        {
            try
            {
                var tokenJson = await SecureStorage.GetAsync(TokenInfoKey);
                if (string.IsNullOrEmpty(tokenJson))
                    return new InvalidResult<bool>("No token info. You are not signed in.");

                var tokenInfo = JsonConvert.DeserializeObject<TokenResponse>(tokenJson);


                using (var client = new HttpClient())
                {
                    var response = await client.PostAsync($"{_baseUrl}/signin/token?client_id={_authSettings.ClientId}&client_secret={_authSettings.ClientSecret}&refresh_token={tokenInfo.RefreshToken}&grant_type=refresh_token&redirect_uri=suavekeystwitch://", null);
                    if (response?.IsSuccessStatusCode == true)
                    {
                        var json = await response.Content.ReadAsStringAsync();
                        tokenInfo = JsonConvert.DeserializeObject<TokenResponse>(json);

                        // we have our tokens. Gotta do something with it
                        _currentToken = tokenInfo?.AccessToken;
                        await StoreTokenInfo(tokenInfo);

                        return new SuccessResult<bool>(true);
                    }
                }
                return new InvalidResult<bool>("Unable to authenticate.");

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
                return new UnexpectedResult<bool>();
            }
        }

        public async Task<Result<bool>> SendCommandAsync(string command)
        {
            try
            {
                var tokenResult = await GetAccessTokenAsync();
                using (var client = new HttpClient())
                {
                    client.DefaultRequestHeaders.Add("Authorization", $"Bearer {tokenResult.Data}");
                    var response = await client.PostAsync($"{_baseUrl}/api/keyboard/KeyPress?key={command}", null);
                    if (response?.IsSuccessStatusCode != true)
                        return new InvalidResult<bool>("Error sending command to suave keys");
                }
                return new SuccessResult<bool>(true);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
                return new UnexpectedResult<bool>();
            }
            
        }

        public async Task<Result<bool>> StartSignInAsync()
        {
            try
            {
                var existingToken = await GetAccessTokenAsync();
                if (existingToken?.ResultType == ResultType.Ok && existingToken.Data != null)
                    return new SuccessResult<bool>(true);

                var state = Guid.NewGuid().ToString();
                var clientId = _authSettings.ClientId;
                var authResult = await WebAuthenticator.AuthenticateAsync(
                    new Uri($"{_baseUrl}/signin?client_id={clientId}&state={state}&redirect_uri=suavekeystwitch://"),
                    new Uri("suavekeystwitch://"));

                var code = authResult.Properties["code"];
                var confirmState = authResult.Properties["state"];

                if (state != confirmState)
                    return new InvalidResult<bool>("Invalid state. This might be a sign of an interception attack.");

                using (var client = new HttpClient())
                {
                    var response = await client.PostAsync($"{_baseUrl}/signin/token?client_id={clientId}&code={code}&grant_type=authorization_code&redirect_uri=suavekeystwitch://", null);
                    if (response?.IsSuccessStatusCode == true)
                    {
                        var json = await response.Content.ReadAsStringAsync();
                        var tokenInfo = JsonConvert.DeserializeObject<TokenResponse>(json);

                        // we have our tokens. Gotta do something with it
                        _currentToken = tokenInfo?.AccessToken;
                        await StoreTokenInfo(tokenInfo);

                        return new SuccessResult<bool>(true);
                    }
                }


                return new InvalidResult<bool>("Unable to authenticate.");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
                return new UnexpectedResult<bool>();
            }
        }

        private async Task StoreTokenInfo(TokenResponse tokenInfo)
        {
            await SecureStorage.SetAsync(TokenInfoKey, JsonConvert.SerializeObject(tokenInfo));
        }
    }
}
